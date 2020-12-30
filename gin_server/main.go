package main

import (
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/mysql"
)

type (
	userModel struct {
		gorm.Model
		Username string `json:"username"`
		Company  string `json:"company"`
		Apikey   string `json:"apikey`
	}
	sampleModel struct {
		gorm.Model
		Tokens   string `json:"tokens"`
		Head     string `json:"head"`
		Tail     string `json:"tail"`
		Relation string `json:"relation"`
	}
)

// gorm.Model 定义
type Model struct {
	ID        uint `gorm:"primary_key"`
	CreatedAt time.Time
	UpdatedAt time.Time
	DeletedAt *time.Time
}

// 指定表名
func (userModel) TableName() string {
	return "user"
}

var db *gorm.DB

// 初始化
func init() {
	var err error
	var constr string
	constr = fmt.Sprintf("%s:%s@(%s:%d)/%s?charset=utf8mb4&parseTime=True&loc=Local", "root", "root", "127.0.0.1", 3306, "rel-server")
	db, err = gorm.Open("mysql", constr)
	if err != nil {
		fmt.Println(err)
		panic("数据库连接失败")
	}

	db.AutoMigrate(&userModel{})
	db.AutoMigrate(&sampleModel{})
}

const (
	JSON_SUCCESS int = 1
	JSON_ERROR   int = 0
)

type apikeyRequest struct {
	ApiID   string `json:"api_id" binding:"required"`
	Company string `json:"company"`
}

// API-KEY 申请
func getapikey(c *gin.Context) {
	var data apikeyRequest
	if err := c.ShouldBindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	var user userModel
	username := data.ApiID
	db.Where("username = ?", username).First(&user)
	if user.ID == 0 {
		apikey := uuid.New()
		user = userModel{Username: data.ApiID, Company: data.Company, Apikey: apikey.String()}
		db.Save(&user)
		c.JSON(http.StatusOK, gin.H{"api_key": apikey.String()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"api_key": user.Apikey})
}

type sample struct {
	Tokens   string `json:"tokens"`
	Head     string `json:"head"`
	Tail     string `json:"tail"`
	Relation string `json:"relation"`
}

// 样本提交
func submitsample(c *gin.Context) {
	var data sample
	if err := c.ShouldBindJSON(&data); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	sample := sampleModel{Tokens: data.Tokens, Head: data.Head, Tail: data.Tail, Relation: data.Relation}
	db.Save(&sample)
	c.JSON(http.StatusOK, gin.H{
		"status": JSON_SUCCESS,
	})
}

// 关系识别
func relationIdentify(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"relation_type": "Component-Whole(e2,e1)",
	})
}

//样本审核
func checksample(c *gin.Context) {
	var sp sampleModel
	db.First(&sp)
	if sp.ID == 0 {
		c.HTML(http.StatusOK, "index.tmpl", gin.H{
			"title": "无剩余样本",
		})
		return
	}
	c.HTML(http.StatusOK, "index.tmpl", gin.H{
		"tokens":   sp.Tokens,
		"head":     sp.Head,
		"tail":     sp.Tail,
		"realtion": sp.Relation,
	})
}

func main() {
	r := gin.Default()
	r.LoadHTMLGlob("templates/*")
	v1 := r.Group("/api/")
	{
		v1.POST("/getapikey", getapikey)               // API-KEY 申请
		v1.POST("/relationidentify", relationIdentify) // 关系识别
		v1.POST("/submitsample", submitsample)         // 样本提交
	}
	r.GET("/checksample", checksample) //样本审核
	r.Run(":9089")
}
